# SpikeATac: A Multimodal Tactile Finger with Taxelized Dynamic Sensing for Dexterous Manipulation 

**Title (ZH)**: SpikeATac: 具有税el化动态传感的多模态触觉手指用于灵巧操作 

**Authors**: Eric T. Chang, Peter Ballentine, Zhanpeng He, Do-Gon Kim, Kai Jiang, Hua-Hsuan Liang, Joaquin Palacios, William Wang, Pedro Piacenza, Ioannis Kymissis, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2510.27048)  

**Abstract**: In this work, we introduce SpikeATac, a multimodal tactile finger combining a taxelized and highly sensitive dynamic response (PVDF) with a static transduction method (capacitive) for multimodal touch sensing. Named for its `spiky' response, SpikeATac's 16-taxel PVDF film sampled at 4 kHz provides fast, sensitive dynamic signals to the very onset and breaking of contact. We characterize the sensitivity of the different modalities, and show that SpikeATac provides the ability to stop quickly and delicately when grasping fragile, deformable objects. Beyond parallel grasping, we show that SpikeATac can be used in a learning-based framework to achieve new capabilities on a dexterous multifingered robot hand. We use a learning recipe that combines reinforcement learning from human feedback with tactile-based rewards to fine-tune the behavior of a policy to modulate force. Our hardware platform and learning pipeline together enable a difficult dexterous and contact-rich task that has not previously been achieved: in-hand manipulation of fragile objects. Videos are available at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 本研究介绍了SpikeATac，一种多模态触觉手指，结合了税el化的动态高灵敏度响应（PVDF）和静态转换方法（电容式）以实现多模态触觉传感。SpikeATac以其“尖峰”响应而命名，其16税el的PVDF膜以4 kHz的频率采样，提供快速敏感的动态信号，用于接触的开始和结束。我们表征了不同模态的灵敏度，并展示了SpikeATac能够在抓取脆弱可变形物体时快速而细致地停止。除了并行抓取，我们还展示了SpikeATac可以在基于学习的框架中用于实现灵巧多指机器人手的新能力。我们使用了一种结合来自人类反馈的强化学习与基于触觉的奖励的学习方法，以微调策略的行为来调节力。我们的硬件平台和学习管道共同使一项以前未达成的复杂灵巧且触觉丰富的任务成为可能：在手内部操作脆弱物体。视频可在\href{this https URL}{this http URL}获取。 

---
# PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting 

**Title (ZH)**: PETAR：基于掩码意识的视觉-语言模型在PET自动化报告中的局部发现生成 

**Authors**: Danyal Maqbool, Changhee Lee, Zachary Huemann, Samuel D. Church, Matthew E. Larson, Scott B. Perlman, Tomas A. Romero, Joshua D. Warner, Meghan Lubner, Xin Tie, Jameson Merkow, Junjie Hu, Steve Y. Cho, Tyler J. Bradshaw  

**Link**: [PDF](https://arxiv.org/pdf/2510.27680)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled impressive multimodal reasoning, yet most medical applications remain limited to 2D imaging. In this work, we extend VLMs to 3D positron emission tomography and computed tomography (PET/CT), a domain characterized by large volumetric data, small and dispersed lesions, and lengthy radiology reports. We introduce a large-scale dataset comprising over 11,000 lesion-level descriptions paired with 3D segmentations from more than 5,000 PET/CT exams, extracted via a hybrid rule-based and large language model (LLM) pipeline. Building upon this dataset, we propose PETAR-4B, a 3D mask-aware vision-language model that integrates PET, CT, and lesion contours for spatially grounded report generation. PETAR bridges global contextual reasoning with fine-grained lesion awareness, producing clinically coherent and localized findings. Comprehensive automated and human evaluations demonstrate that PETAR substantially improves PET/CT report generation quality, advancing 3D medical vision-language understanding. 

**Abstract (ZH)**: 近期视觉语言模型的发展已在多模态推理方面取得了显著进展，但大多数医疗应用仍然局限于2D成像。本文将视觉语言模型扩展至正电子发射断层扫描和计算机断层扫描(PET/CT)，该领域以大型体数据、小且分散的病灶以及冗长的放射学报告为特征。我们引入了一个大型数据集，包含超过11,000个病灶级别的描述，以及来自超过5,000次PET/CT检查的3D分割，通过混合基于规则和大型语言模型的管道提取。基于该数据集，我们提出了PETAR-4B，这是一种3D掩码感知的视觉语言模型，能够融合PET、CT和病灶轮廓，以实现空间定位的报告生成。PETAR将全局上下文推理与细粒度的病灶意识相结合，生成临床一致且局部化的发现。全面的自动化和人工评估表明，PETAR显著提高了PET/CT报告生成的质量，推动了3D医疗视觉语言理解的进步。 

---
# Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation 

**Title (ZH)**: 草图引导的多模态布局生成 

**Authors**: Riccardo Brioschi, Aleksandr Alekseev, Emanuele Nevali, Berkay Döner, Omar El Malki, Blagoj Mitrevski, Leandro Kieliger, Mark Collier, Andrii Maksai, Jesse Berent, Claudiu Musat, Efi Kokiopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2510.27632)  

**Abstract**: Graphic layout generation is a growing research area focusing on generating aesthetically pleasing layouts ranging from poster designs to documents. While recent research has explored ways to incorporate user constraints to guide the layout generation, these constraints often require complex specifications which reduce usability. We introduce an innovative approach exploiting user-provided sketches as intuitive constraints and we demonstrate empirically the effectiveness of this new guidance method, establishing the sketch-to-layout problem as a promising research direction, which is currently under-explored. To tackle the sketch-to-layout problem, we propose a multimodal transformer-based solution using the sketch and the content assets as inputs to produce high quality layouts. Since collecting sketch training data from human annotators to train our model is very costly, we introduce a novel and efficient method to synthetically generate training sketches at scale. We train and evaluate our model on three publicly available datasets: PubLayNet, DocLayNet and SlidesVQA, demonstrating that it outperforms state-of-the-art constraint-based methods, while offering a more intuitive design experience. In order to facilitate future sketch-to-layout research, we release O(200k) synthetically-generated sketches for the public datasets above. The datasets are available at this https URL. 

**Abstract (ZH)**: 图形布局生成是一个快速增长的研究领域，专注于从海报设计到文档等各种 aesthetically pleasing 布局的生成。尽管近期的研究探讨了通过引入用户约束来指导布局生成的方法，但这些约束往往需要复杂的规格说明，从而降低了 usability。我们提出了一种创新的方法，利用用户提供草图作为直观的约束条件，并通过实验证明了这种方法的有效性，将草图转换为布局的问题确立为一个有前景的研究方向，目前这一方向尚未得到充分探索。为了应对草图到布局的问题，我们提出了一种基于多模态变压器的解决方案，使用草图和内容资产作为输入以生成高质量的布局。由于从人类标注者收集用于训练模型的草图训练数据成本高昂，我们引入了一种新颖且高效的方法来大规模合成训练草图。我们使用公开可用的三个数据集：PubLayNet、DocLayNet 和 SlidesVQA，对模型进行训练和评估，证明该方法比现有的基于约束的方法表现更优，同时提供了更直观的设计体验。为了便于未来的草图到布局研究，我们发布了超过20万张合成生成的草图，供上述公开数据集使用。数据集可通过以下链接获取。 

---
# Context-Gated Cross-Modal Perception with Visual Mamba for PET-CT Lung Tumor Segmentation 

**Title (ZH)**: 基于上下文门控跨模态感知的视觉Mamba PET-CT肺肿瘤分割 

**Authors**: Elena Mulero Ayllón, Linlin Shen, Pierangelo Veltri, Fabrizia Gelardi, Arturo Chiti, Paolo Soda, Matteo Tortora  

**Link**: [PDF](https://arxiv.org/pdf/2510.27508)  

**Abstract**: Accurate lung tumor segmentation is vital for improving diagnosis and treatment planning, and effectively combining anatomical and functional information from PET and CT remains a major challenge. In this study, we propose vMambaX, a lightweight multimodal framework integrating PET and CT scan images through a Context-Gated Cross-Modal Perception Module (CGM). Built on the Visual Mamba architecture, vMambaX adaptively enhances inter-modality feature interaction, emphasizing informative regions while suppressing noise. Evaluated on the PCLT20K dataset, the model outperforms baseline models while maintaining lower computational complexity. These results highlight the effectiveness of adaptive cross-modal gating for multimodal tumor segmentation and demonstrate the potential of vMambaX as an efficient and scalable framework for advanced lung cancer analysis. The code is available at this https URL. 

**Abstract (ZH)**: 准确的肺部肿瘤分割对于提高诊断和治疗规划至关重要，有效结合PET和CT的解剖与功能信息仍然是一个主要挑战。本研究提出了一种轻量级多模态框架vMambaX，该框架通过上下文门控跨模态感知模块(CGM)整合PET和CT扫描图像。基于Visual Mamba架构，vMambaX自适应地增强跨模态特征交互，强调信息丰富的区域并抑制噪声。在PCLT20K数据集上的评估结果显示，该模型在保持较低计算复杂度的同时优于基准模型。这些结果突显了自适应跨模态门控在多模态肿瘤分割中的有效性，并展示了vMambaX作为高级肺癌分析高效且可扩展框架的潜力。代码可在以下链接获取：this https URL。 

---
# FOCUS: Efficient Keyframe Selection for Long Video Understanding 

**Title (ZH)**: FOCUS: 长视频理解中的高效关键帧选择 

**Authors**: Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2510.27280)  

**Abstract**: Multimodal large language models (MLLMs) represent images and video frames as visual tokens. Scaling from single images to hour-long videos, however, inflates the token budget far beyond practical limits. Popular pipelines therefore either uniformly subsample or apply keyframe selection with retrieval-style scoring using smaller vision-language models. However, these keyframe selection methods still rely on pre-filtering before selection to reduce the inference cost and can miss the most informative moments.
We propose FOCUS, Frame-Optimistic Confidence Upper-bound Selection, a training-free, model-agnostic keyframe selection module that selects query-relevant frames under a strict token budget. FOCUS formulates keyframe selection as a combinatorial pure-exploration (CPE) problem in multi-armed bandits: it treats short temporal clips as arms, and uses empirical means and Bernstein confidence radius to identify informative regions while preserving exploration of uncertain areas. The resulting two-stage exploration-exploitation procedure reduces from a sequential policy with theoretical guarantees, first identifying high-value temporal regions, then selecting top-scoring frames within each region On two long-video question-answering benchmarks, FOCUS delivers substantial accuracy improvements while processing less than 2% of video frames. For videos longer than 20 minutes, it achieves an 11.9% gain in accuracy on LongVideoBench, demonstrating its effectiveness as a keyframe selection method and providing a simple and general solution for scalable long-video understanding with MLLMs. 

**Abstract (ZH)**: 基于帧优化置信上界选择的多模态大型语言模型关键帧选择模块 

---
# Multi-Modal Feature Fusion for Spatial Morphology Analysis of Traditional Villages via Hierarchical Graph Neural Networks 

**Title (ZH)**: 基于层次图神经网络的多模态特征融合传统 villages 空间形态分析 

**Authors**: Jiaxin Zhang, Zehong Zhu, Junye Deng, Yunqin Li, and Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27208)  

**Abstract**: Villages areas hold significant importance in the study of human-land relationships. However, with the advancement of urbanization, the gradual disappearance of spatial characteristics and the homogenization of landscapes have emerged as prominent issues. Existing studies primarily adopt a single-disciplinary perspective to analyze villages spatial morphology and its influencing factors, relying heavily on qualitative analysis methods. These efforts are often constrained by the lack of digital infrastructure and insufficient data. To address the current research limitations, this paper proposes a Hierarchical Graph Neural Network (HGNN) model that integrates multi-source data to conduct an in-depth analysis of villages spatial morphology. The framework includes two types of nodes-input nodes and communication nodes-and two types of edges-static input edges and dynamic communication edges. By combining Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), the proposed model efficiently integrates multimodal features under a two-stage feature update mechanism. Additionally, based on existing principles for classifying villages spatial morphology, the paper introduces a relational pooling mechanism and implements a joint training strategy across 17 subtypes. Experimental results demonstrate that this method achieves significant performance improvements over existing approaches in multimodal fusion and classification tasks. Additionally, the proposed joint optimization of all sub-types lifts mean accuracy/F1 from 0.71/0.83 (independent models) to 0.82/0.90, driven by a 6% gain for parcel tasks. Our method provides scientific evidence for exploring villages spatial patterns and generative logic. 

**Abstract (ZH)**: 农村地区在人类-土地关系研究中具有重要意义。然而，随着城市化进程的推进，空间特征的逐渐消失和景观的同质化成为了突出问题。现有研究主要从单一学科视角分析村庄的空间形态及其影响因素，依赖于定性分析方法。这些努力往往受限于缺乏数字基础设施和数据不足的问题。为了解决现有研究的局限性，本文提出了一种层次图神经网络（Hierarchical Graph Neural Network, HGNN）模型，该模型整合多源数据以深入分析村庄的空间形态。该框架包括输入节点和通信节点两种类型，以及静态输入边和动态通信边两种类型。通过结合图卷积网络（Graph Convolutional Networks, GCN）和图注意网络（Graph Attention Networks, GAT），所提出的模型在两阶段特征更新机制下高效地整合了多模态特征。此外，基于现有的村庄空间形态分类原则，本文引入了关系聚池化机制，并通过综合培训策略实施了17种亚型间的联合训练。实验结果表明，该方法在多模态融合和分类任务中相比现有方法取得了显著性能提升。此外，对所有亚型的联合优化进一步将平均准确度/F1分数从独立模型的0.71/0.83提升至0.82/0.90， parcel任务提高了6%。该方法为探索村庄空间模式和生成逻辑提供了科学依据。 

---
