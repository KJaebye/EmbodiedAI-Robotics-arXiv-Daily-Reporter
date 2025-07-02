# Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation 

**Title (ZH)**: 带有正弦编码的多模态图卷积网络在鲁棒人体动作分割中的应用 

**Authors**: Hao Xing, Kai Zhe Boey, Yuankai Wu, Darius Burschka, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00752)  

**Abstract**: Accurate temporal segmentation of human actions is critical for intelligent robots in collaborative settings, where a precise understanding of sub-activity labels and their temporal structure is essential. However, the inherent noise in both human pose estimation and object detection often leads to over-segmentation errors, disrupting the coherence of action sequences. To address this, we propose a Multi-Modal Graph Convolutional Network (MMGCN) that integrates low-frame-rate (e.g., 1 fps) visual data with high-frame-rate (e.g., 30 fps) motion data (skeleton and object detections) to mitigate fragmentation. Our framework introduces three key contributions. First, a sinusoidal encoding strategy that maps 3D skeleton coordinates into a continuous sin-cos space to enhance spatial representation robustness. Second, a temporal graph fusion module that aligns multi-modal inputs with differing resolutions via hierarchical feature aggregation, Third, inspired by the smooth transitions inherent to human actions, we design SmoothLabelMix, a data augmentation technique that mixes input sequences and labels to generate synthetic training examples with gradual action transitions, enhancing temporal consistency in predictions and reducing over-segmentation artifacts.
Extensive experiments on the Bimanual Actions Dataset, a public benchmark for human-object interaction understanding, demonstrate that our approach outperforms state-of-the-art methods, especially in action segmentation accuracy, achieving F1@10: 94.5% and F1@25: 92.8%. 

**Abstract (ZH)**: 多模态图卷积网络在人体动作时间分割中的应用：一种集成低帧率视觉数据与高帧率motion数据的方法 

---
# Audio-3DVG: Unified Audio - Point Cloud Fusion for 3D Visual Grounding 

**Title (ZH)**: Audio-3DVG：统一的音频-点云融合三维视觉定位 

**Authors**: Duc Cao-Dinh, Khai Le-Duc, Anh Dao, Bach Phan Tat, Chris Ngo, Duy M. H. Nguyen, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00669)  

**Abstract**: 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an Audio-Guided Attention module that captures interactions between candidate objects and relational speech cues, improving target discrimination in cluttered scenes. To support benchmarking, we synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods-highlighting the promise of integrating spoken language into 3D vision tasks. 

**Abstract (ZH)**: 基于音频的三维视觉定位（Audio-3D Visual Grounding）：一种结合音频和空间信息的简单有效框架 

---
# VoyagerVision: Investigating the Role of Multi-modal Information for Open-ended Learning Systems 

**Title (ZH)**: VoyagerVision：探索多模态信息在开放学习系统中的作用 

**Authors**: Ethan Smyth, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2507.00079)  

**Abstract**: Open-endedness is an active field of research in the pursuit of capable Artificial General Intelligence (AGI), allowing models to pursue tasks of their own choosing. Simultaneously, recent advancements in Large Language Models (LLMs) such as GPT-4o [9] have allowed such models to be capable of interpreting image inputs. Implementations such as OMNI-EPIC [4] have made use of such features, providing an LLM with pixel data of an agent's POV to parse the environment and allow it to solve tasks. This paper proposes that providing these visual inputs to a model gives it greater ability to interpret spatial environments, and as such, can increase the number of tasks it can successfully perform, extending its open-ended potential. To this aim, this paper proposes VoyagerVision -- a multi-modal model capable of creating structures within Minecraft using screenshots as a form of visual feedback, building on the foundation of Voyager. VoyagerVision was capable of creating an average of 2.75 unique structures within fifty iterations of the system, as Voyager was incapable of this, it is an extension in an entirely new direction. Additionally, in a set of building unit tests VoyagerVision was successful in half of all attempts in flat worlds, with most failures arising in more complex structures. Project website is available at this https URL 

**Abstract (ZH)**: 开放性是追求具备通用人工智能（AGI）能力的研究中的一个活跃领域，使模型能够自主追求任务。同时，大型语言模型（LLMs）如GPT-4o的近期进展使其能够解释图像输入。OMNI-EPIC等实现使用了这些功能，为LLM提供代理视角的像素数据以解析环境，并允许其解决任务。本文提出，向模型提供这些视觉输入使其能够更好地解释空间环境，从而增加其能够成功完成的任务数量，扩展其开放性潜力。为此，本文提出了VoyagerVision——一种多模态模型，能够使用截屏作为视觉反馈在Minecraft中创建结构，建立在Voyager的基础上。VoyagerVision在系统五十次迭代中平均能够创建2.75种独特的结构，而Voyager无法做到这一点，因此它是朝完全新方向的拓展。此外，在一组建筑单元测试中，VoyagerVision在平坦世界中成功了一半的尝试，大多数失败发生在更复杂的结构中。项目网站可在以下链接访问。 

---
# DiMo-GUI: Advancing Test-time Scaling in GUI Grounding via Modality-Aware Visual Reasoning 

**Title (ZH)**: DiMo-GUI: 基于模态意识视觉推理的GUI定位测试时扩展增强 

**Authors**: Hang Wu, Hongkai Chen, Yujun Cai, Chang Liu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00008)  

**Abstract**: Grounding natural language queries in graphical user interfaces (GUIs) poses unique challenges due to the diversity of visual elements, spatial clutter, and the ambiguity of language. In this paper, we introduce DiMo-GUI, a training-free framework for GUI grounding that leverages two core strategies: dynamic visual grounding and modality-aware optimization. Instead of treating the GUI as a monolithic image, our method splits the input into textual elements and iconic elements, allowing the model to reason over each modality independently using general-purpose vision-language models. When predictions are ambiguous or incorrect, DiMo-GUI dynamically focuses attention by generating candidate focal regions centered on the model's initial predictions and incrementally zooms into subregions to refine the grounding result. This hierarchical refinement process helps disambiguate visually crowded layouts without the need for additional training or annotations. We evaluate our approach on standard GUI grounding benchmarks and demonstrate consistent improvements over baseline inference pipelines, highlighting the effectiveness of combining modality separation with region-focused reasoning. 

**Abstract (ZH)**: 基于图形用户界面（GUI）的自然语言查询 grounding 面临着独特挑战，由于视觉元素的多样性、空间重叠以及语言的歧义性。本文介绍了一种无需训练的 GUI grounding 框架 DiMo-GUI，该框架采用了动态视觉 grounding 和模态感知优化两种核心策略。我们的方法将输入分为文本元素和图示元素，允许模型使用通用的视觉-语言模型独立推理每种模态。当预测结果模糊或错误时，DiMo-GUI 会动态聚焦注意力，生成以模型初始预测为中心的候选焦点区域，并逐步放大子区域以细化 grounding 结果。这一分层细化过程有助于在无需额外训练或标注的情况下消解视觉拥挤的布局。我们在标准 GUI grounding 数据集上评估了该方法，展示了与基准推理管道相比的一致性改进，突显了结合模态分离与区域集中推理的有效性。 

---
# MemeCMD: An Automatically Generated Chinese Multi-turn Dialogue Dataset with Contextually Retrieved Memes 

**Title (ZH)**: MemeCMD：一个基于上下文检索的表情包自动生成的中文多轮对话数据集 

**Authors**: Yuheng Wang, Xianhe Tang, Pufeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00891)  

**Abstract**: Memes are widely used in online social interactions, providing vivid, intuitive, and often humorous means to express intentions and emotions. Existing dialogue datasets are predominantly limited to either manually annotated or pure-text conversations, lacking the expressiveness and contextual nuance that multimodal interactions this http URL address these challenges, we introduce MemeCMD, an automatically generated Chinese Multi-turn Dialogue dataset with contextually retrieved memes. Our dataset combines a large-scale, MLLM-annotated meme library with dialogues auto-generated by dual agents across diverse scenarios. We introduce a retrieval framework and adaptive threshold to ensure contextually relevant, naturally spaced meme usage. Experiments demonstrate the effectiveness of our approach in generating contextually appropriate and diverse meme-incorporated dialogues, offering a scalable and privacy-preserving resource for advancing multimodal conversational AI. 

**Abstract (ZH)**: MEME-CMD：一个基于上下文检索的自动生成多轮中文对话数据集 

---
# ATSTrack: Enhancing Visual-Language Tracking by Aligning Temporal and Spatial Scales 

**Title (ZH)**: ATSTrack: 通过对齐时间和空间尺度 Enhance 视觉-语言跟踪 

**Authors**: Yihao Zhen, Qiang Wang, Yu Qiao, Liangqiong Qu, Huijie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00454)  

**Abstract**: A main challenge of Visual-Language Tracking (VLT) is the misalignment between visual inputs and language descriptions caused by target movement. Previous trackers have explored many effective feature modification methods to preserve more aligned features. However, an important yet unexplored factor ultimately hinders their capability, which is the inherent differences in the temporal and spatial scale of information between visual and language inputs. To address this issue, we propose a novel visual-language tracker that enhances the effect of feature modification by \textbf{A}ligning \textbf{T}emporal and \textbf{S}patial scale of different input components, named as \textbf{ATSTrack}. Specifically, we decompose each language description into phrases with different attributes based on their temporal and spatial correspondence with visual inputs, and modify their features in a fine-grained manner. Moreover, we introduce a Visual-Language token that comprises modified linguistic information from the previous frame to guide the model to extract visual features that are more relevant to language description, thereby reducing the impact caused by the differences in spatial scale. Experimental results show that our proposed ATSTrack achieves performance comparable to existing methods. Our code will be released. 

**Abstract (ZH)**: 视觉-语言跟踪中的主要挑战是目标移动导致的视觉输入与语言描述之间的对齐不一致。为了应对这一问题，我们提出了一种新颖的视觉-语言跟踪器ATSTrack，通过对齐不同输入组件的时空尺度来增强特征修改的效果。具体地，我们根据语言描述与视觉输入的时空对应关系，将每个语言描述分解为具有不同属性的短语，并以精细的方式修改其特征。此外，我们引入了一个视觉-语言令牌，该令牌包含来自上一帧的修改语言信息，以引导模型提取与语言描述更相关的视觉特征，从而减少由于时空尺度差异引起的影响。实验结果表明，提出的ATSTrack在性能上与现有方法相当。我们的代码将公开发布。 

---
# Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM) 

**Title (ZH)**: 多模态多疾病医疗影像基础模型（MerMED-FM） 

**Authors**: Yang Zhou, Chrystie Wan Ning Quek, Jun Zhou, Yan Wang, Yang Bai, Yuhe Ke, Jie Yao, Laura Gutierrez, Zhen Ling Teo, Darren Shu Jeng Ting, Brian T. Soetikno, Christopher S. Nielsen, Tobias Elze, Zengxiang Li, Linh Le Dinh, Lionel Tim-Ee Cheng, Tran Nguyen Tuan Anh, Chee Leong Cheng, Tien Yin Wong, Nan Liu, Iain Beehuat Tan, Tony Kiat Hon Lim, Rick Siow Mong Goh, Yong Liu, Daniel Shu Wei Ting  

**Link**: [PDF](https://arxiv.org/pdf/2507.00185)  

**Abstract**: Current artificial intelligence models for medical imaging are predominantly single modality and single disease. Attempts to create multimodal and multi-disease models have resulted in inconsistent clinical accuracy. Furthermore, training these models typically requires large, labour-intensive, well-labelled datasets. We developed MerMED-FM, a state-of-the-art multimodal, multi-specialty foundation model trained using self-supervised learning and a memory module. MerMED-FM was trained on 3.3 million medical images from over ten specialties and seven modalities, including computed tomography (CT), chest X-rays (CXR), ultrasound (US), pathology patches, color fundus photography (CFP), optical coherence tomography (OCT) and dermatology images. MerMED-FM was evaluated across multiple diseases and compared against existing foundational models. Strong performance was achieved across all modalities, with AUROCs of 0.988 (OCT); 0.982 (pathology); 0.951 (US); 0.943 (CT); 0.931 (skin); 0.894 (CFP); 0.858 (CXR). MerMED-FM has the potential to be a highly adaptable, versatile, cross-specialty foundation model that enables robust medical imaging interpretation across diverse medical disciplines. 

**Abstract (ZH)**: 当前的医疗成像人工智能模型主要为单模态和单疾病模型。尝试创建多模态和多疾病模型在临床准确性上表现出不一致的结果。此外，训练这些模型通常需要大量的、劳动密集型的、高质量标注的数据集。我们开发了MerMED-FM，这是一种基于自我监督学习和记忆模块训练的最先进的多模态、多专科基础模型。MerMED-FM基于超过十种专科和七种模态的330万份医疗图像进行训练，包括CT、胸片（CXR）、超声（US）、病理切片、彩色眼底摄影（CFP）、光学相干断层扫描（OCT）和皮肤科图像。MerMED-FM在多种疾病上进行了评估，并与现有基础模型进行了对比。在所有模态中均取得了优异表现，AUROCs分别为：OCT，0.988；病理，0.982；超声，0.951；CT，0.943；皮肤，0.931；CFP，0.894；CXR，0.858。MerMED-FM有望成为一种高度适应性强、多功能且跨专科的基础模型，能够跨多种医学领域实现稳健的医疗影像解释。 

---
# MANTA: Cross-Modal Semantic Alignment and Information-Theoretic Optimization for Long-form Multimodal Understanding 

**Title (ZH)**: MANTA：跨模态语义对齐与信息论优化以实现长形式多模态理解 

**Authors**: Ziqi Zhong, Daniel Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00068)  

**Abstract**: While multi-modal learning has advanced significantly, current approaches often treat modalities separately, creating inconsistencies in representation and reasoning. We introduce MANTA (Multi-modal Abstraction and Normalization via Textual Alignment), a theoretically-grounded framework that unifies visual and auditory inputs into a structured textual space for seamless processing with large language models. MANTA addresses four key challenges: (1) semantic alignment across modalities with information-theoretic optimization, (2) adaptive temporal synchronization for varying information densities, (3) hierarchical content representation for multi-scale understanding, and (4) context-aware retrieval of sparse information from long sequences. We formalize our approach within a rigorous mathematical framework, proving its optimality for context selection under token constraints. Extensive experiments on the challenging task of Long Video Question Answering show that MANTA improves state-of-the-art models by up to 22.6% in overall accuracy, with particularly significant gains (27.3%) on videos exceeding 30 minutes. Additionally, we demonstrate MANTA's superiority on temporal reasoning tasks (23.8% improvement) and cross-modal understanding (25.1% improvement). Our framework introduces novel density estimation techniques for redundancy minimization while preserving rare signals, establishing new foundations for unifying multimodal representations through structured text. 

**Abstract (ZH)**: 多模态抽象和归一化通过文本对齐 

---
