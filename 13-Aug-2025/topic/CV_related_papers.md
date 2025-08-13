# STELAR-VISION: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision 

**Title (ZH)**: STELAR-VISION：自我拓扑意识高效学习用于视觉对齐推理 

**Authors**: Chen Li, Han Zhang, Zhantao Yang, Fangyi Chen, Zihan Wang, Anudeepsekhar Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2508.08688)  

**Abstract**: Vision-language models (VLMs) have made significant strides in reasoning, yet they often struggle with complex multimodal tasks and tend to generate overly verbose outputs. A key limitation is their reliance on chain-of-thought (CoT) reasoning, despite many tasks benefiting from alternative topologies like trees or graphs. To address this, we introduce STELAR-Vision, a training framework for topology-aware reasoning. At its core is TopoAug, a synthetic data pipeline that enriches training with diverse topological structures. Using supervised fine-tuning and reinforcement learning, we post-train Qwen2VL models with both accuracy and efficiency in mind. Additionally, we propose Frugal Learning, which reduces output length with minimal accuracy loss. On MATH-V and VLM-S2H, STELAR-Vision improves accuracy by 9.7% over its base model and surpasses the larger Qwen2VL-72B-Instruct by 7.3%. On five out-of-distribution benchmarks, it outperforms Phi-4-Multimodal-Instruct by up to 28.4% and LLaMA-3.2-11B-Vision-Instruct by up to 13.2%, demonstrating strong generalization. Compared to Chain-Only training, our approach achieves 4.3% higher overall accuracy on in-distribution datasets and consistently outperforms across all OOD benchmarks. We have released datasets, and code will be available. 

**Abstract (ZH)**: Vision-语言模型（VLMs）在推理方面取得了显著进展，但往往在处理复杂多模态任务时表现不佳，倾向于生成冗长的输出。一个关键限制在于它们依赖于链式思考（CoT）推理，尽管许多任务可以从树状或图形等替代拓扑结构中受益。为了解决这个问题，我们引入了STELAR-Vision，这是一种拓扑感知推理的训练框架。其核心是TopoAug，一种合成数据管道，用于通过多样化拓扑结构丰富训练。利用监督微调和强化学习，我们针对Qwen2VL模型进行了后训练，注重准确性和效率。此外，我们提出了节俭学习，以最小的准确率损失减少输出长度。STELAR-Vision在MATH-V和VLM-S2H上的准确率提高了9.7%，并且在基准测试中超过了更大的Qwen2VL-72B-Instruct 7.3%。在五个分布外基准测试中，STELAR-Vision分别比Phi-4-Multimodal-Instruct和LLaMA-3.2-11B-Vision-Instruct提高了最多28.4%和13.2%的准确率，展示了强大的泛化能力。与仅链式训练相比，在分布内数据集上我们的方法整体准确率提高了4.3%，并且在所有分布外基准测试中均表现更优。我们已经发布了相关数据集，并将提供代码。 

---
# Training-Free Text-Guided Color Editing with Multi-Modal Diffusion Transformer 

**Title (ZH)**: 基于多模态扩散变换器的无训练文本引导颜色编辑 

**Authors**: Zixin Yin, Xili Dai, Ling-Hao Chen, Deyu Zhou, Jianan Wang, Duomin Wang, Gang Yu, Lionel M. Ni, Heung-Yeung Shum  

**Link**: [PDF](https://arxiv.org/pdf/2508.09131)  

**Abstract**: Text-guided color editing in images and videos is a fundamental yet unsolved problem, requiring fine-grained manipulation of color attributes, including albedo, light source color, and ambient lighting, while preserving physical consistency in geometry, material properties, and light-matter interactions. Existing training-free methods offer broad applicability across editing tasks but struggle with precise color control and often introduce visual inconsistency in both edited and non-edited regions. In this work, we present ColorCtrl, a training-free color editing method that leverages the attention mechanisms of modern Multi-Modal Diffusion Transformers (MM-DiT). By disentangling structure and color through targeted manipulation of attention maps and value tokens, our method enables accurate and consistent color editing, along with word-level control of attribute intensity. Our method modifies only the intended regions specified by the prompt, leaving unrelated areas untouched. Extensive experiments on both SD3 and FLUX.1-dev demonstrate that ColorCtrl outperforms existing training-free approaches and achieves state-of-the-art performances in both edit quality and consistency. Furthermore, our method surpasses strong commercial models such as FLUX.1 Kontext Max and GPT-4o Image Generation in terms of consistency. When extended to video models like CogVideoX, our approach exhibits greater advantages, particularly in maintaining temporal coherence and editing stability. Finally, our method also generalizes to instruction-based editing diffusion models such as Step1X-Edit and FLUX.1 Kontext dev, further demonstrating its versatility. 

**Abstract (ZH)**: 基于文本指导的图像和视频颜色编辑是一个基础但尚未解决的问题，需要精细操控包括反射率、光源颜色和环境光在内的颜色属性，同时保持几何、材料属性和光物质相互作用的物理一致性。现有的无需训练的方法在编辑任务上具有广泛的应用性，但难以实现精确的颜色控制，并且常常在编辑区域和非编辑区域引入视觉不一致性。在本文中，我们提出了一种无需训练的颜色编辑方法ColorCtrl，该方法利用了现代多模态扩散变换器（MM-DiT）的注意力机制。通过对注意力图和值令牌进行针对性的操控，以分离结构和颜色，我们的方法能够实现准确且一致的颜色编辑，并且具备词级控制属性强度的能力。我们的方法仅修改由提示指定的预期区域，而不影响无关区域。在SD3和FLUX.1-dev上的广泛实验表明，ColorCtrl在编辑质量和一致性方面均优于现有无需训练的方法，并达到了最先进的性能。此外，与强商业模型FLUX.1 Kontext Max和GPT-4o Image Generation相比，我们的方法在一致性方面表现更优。将该方法扩展到视频模型CogVideoX时，我们的方法在保持时序连贯性和编辑稳定性方面表现出更大的优势。最后，我们的方法还适用于基于指令的编辑扩散模型Step1X-Edit和FLUX.1 Kontext dev，进一步展示了其 versatility。 

---
# When Deepfakes Look Real: Detecting AI-Generated Faces with Unlabeled Data due to Annotation Challenges 

**Title (ZH)**: 当深度伪造看起来很真实：由于标注挑战使用未标注数据检测AI生成的脸部图像 

**Authors**: Zhiqiang Yang, Renshuai Tao, Xiaolong Zheng, Guodong Yang, Chunjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09022)  

**Abstract**: Existing deepfake detection methods heavily depend on labeled training data. However, as AI-generated content becomes increasingly realistic, even \textbf{human annotators struggle to distinguish} between deepfakes and authentic images. This makes the labeling process both time-consuming and less reliable. Specifically, there is a growing demand for approaches that can effectively utilize large-scale unlabeled data from online social networks. Unlike typical unsupervised learning tasks, where categories are distinct, AI-generated faces closely mimic real image distributions and share strong similarities, causing performance drop in conventional strategies. In this paper, we introduce the Dual-Path Guidance Network (DPGNet), to tackle two key challenges: (1) bridging the domain gap between faces from different generation models, and (2) utilizing unlabeled image samples. The method features two core modules: text-guided cross-domain alignment, which uses learnable prompts to unify visual and textual embeddings into a domain-invariant feature space, and curriculum-driven pseudo label generation, which dynamically exploit more informative unlabeled samples. To prevent catastrophic forgetting, we also facilitate bridging between domains via cross-domain knowledge distillation. Extensive experiments on \textbf{11 popular datasets}, show that DPGNet outperforms SoTA approaches by \textbf{6.3\%}, highlighting its effectiveness in leveraging unlabeled data to address the annotation challenges posed by the increasing realism of deepfakes. 

**Abstract (ZH)**: 现有深度伪造检测方法 heavily 依赖于标注训练数据，人工智能生成内容日益逼真，人类标注内容难以区分深度伪造和真实图像。这一标注过程既耗时时间又不可靠。具体而言，存在对大型在线社交网络无标注图像样本利用的需求。与典型的无监督学习任务不同，人工智能生成的面孔与真实图像分布高度相似且存在强烈相似性给传统方法带来挑战。本文提出了双引导卷积网络（DPGNet）以应对两个关键挑战：（1）缩小生成模型与真实图像之间的差距；（得起利用无标注图像样本。该方法包含两个关键模块：引导领域对齐模块，通过引导提示将视觉和文本嵌入统一到一个领域不变模态；以及渐进式伪标注生成模块，动态利用更具信息量的无标注样本。为防止灾难性遗忘现象我们还也促荐域知识蒸馏以促进领域之间的知识转移。实验结果表明DPGNet在多个流行数据集上上超越了现有方法6.3%表现其在利用无标注数据应对应挑战深度伪造日益逼真的问题上的有效性。 

---
# Geometry-Aware Global Feature Aggregation for Real-Time Indirect Illumination 

**Title (ZH)**: 基于几何感知的全局特征聚合方法用于实时间接光照 

**Authors**: Meng Gai, Guoping Wang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08826)  

**Abstract**: Real-time rendering with global illumination is crucial to afford the user realistic experience in virtual environments. We present a learning-based estimator to predict diffuse indirect illumination in screen space, which then is combined with direct illumination to synthesize globally-illuminated high dynamic range (HDR) results. Our approach tackles the challenges of capturing long-range/long-distance indirect illumination when employing neural networks and is generalized to handle complex lighting and scenarios.
From the neural network thinking of the solver to the rendering equation, we present a novel network architecture to predict indirect illumination. Our network is equipped with a modified attention mechanism that aggregates global information guided by spacial geometry features, as well as a monochromatic design that encodes each color channel individually.
We conducted extensive evaluations, and the experimental results demonstrate our superiority over previous learning-based techniques. Our approach excels at handling complex lighting such as varying-colored lighting and environment lighting. It can successfully capture distant indirect illumination and simulates the interreflections between textured surfaces well (i.e., color bleeding effects); it can also effectively handle new scenes that are not present in the training dataset. 

**Abstract (ZH)**: 基于学习的屏幕空间间接光照预测方法在实时渲染全局光照中对于提供虚拟环境中的真实体验至关重要。我们提出了一种学习驱动的估计器来预测屏幕空间中的漫反射间接光照，然后将其与直接光照结合以合成全局光照下的高动态范围（HDR）结果。我们的方法解决了在使用神经网络捕捉远距离间接光照时的挑战，并能够处理复杂的光照和场景。从求解器的神经网络思维到渲染方程，我们提出了一种新的网络架构来预测间接光照。我们的网络配备了一种改进的注意力机制，该机制通过空间几何特征聚合全局信息，并且采用单色设计，分别编码每个颜色通道。我们进行了广泛评估，实验结果表明我们的方法在复杂的光照条件下（如变色光照和环境光照）优于之前的基于学习的方法。我们的方法能够捕捉远处的间接光照，模拟纹理表面之间的互反射效果（即颜色泄露效果），并且能够有效处理未包含在训练数据集中的新场景。 

---
# Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation 

**Title (ZH)**: 搭建桥梁：一种基于社会网络压缩仿真的人类世界视频换脸检测框架 

**Authors**: Andrea Montibeller, Dasara Shullani, Daniele Baracchi, Alessandro Piva, Giulia Boato  

**Link**: [PDF](https://arxiv.org/pdf/2508.08765)  

**Abstract**: The growing presence of AI-generated videos on social networks poses new challenges for deepfake detection, as detectors trained under controlled conditions often fail to generalize to real-world scenarios. A key factor behind this gap is the aggressive, proprietary compression applied by platforms like YouTube and Facebook, which launder low-level forensic cues. However, replicating these transformations at scale is difficult due to API limitations and data-sharing constraints. For these reasons, we propose a first framework that emulates the video sharing pipelines of social networks by estimating compression and resizing parameters from a small set of uploaded videos. These parameters enable a local emulator capable of reproducing platform-specific artifacts on large datasets without direct API access. Experiments on FaceForensics++ videos shared via social networks demonstrate that our emulated data closely matches the degradation patterns of real uploads. Furthermore, detectors fine-tuned on emulated videos achieve comparable performance to those trained on actual shared media. Our approach offers a scalable and practical solution for bridging the gap between lab-based training and real-world deployment of deepfake detectors, particularly in the underexplored domain of compressed video content. 

**Abstract (ZH)**: 社交媒体平台上AI生成视频的增加为深度伪造检测带来了新挑战，由于平台如YouTube和Facebook应用的强烈专有压缩方式掩饰了低级鉴伪线索，现有的在受控条件下训练的检测器往往难以在真实世界场景中泛化。因此，我们提出了一个框架，通过估计一小组上传视频的压缩和缩放参数来模拟社交媒体的视频分享管道，从而使不需要直接使用API即可在大数据集上重现平台特定的伪造痕迹。实验表明，通过我们模拟的数据训练的检测器在性能上与使用实际共享媒体训练的检测器相当。该方法提供了一种可扩展且实用的解决方案，用于弥合实验室训练与实际部署深度伪造检测器之间的差距，尤其是在压缩视频内容这一未充分探索的领域。 

---
# SafeFix: Targeted Model Repair via Controlled Image Generation 

**Title (ZH)**: SafeFix：通过可控图像生成的目标模型修复 

**Authors**: Ouyang Xu, Baoming Zhang, Ruiyu Mao, Yunhui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08701)  

**Abstract**: Deep learning models for visual recognition often exhibit systematic errors due to underrepresented semantic subpopulations. Although existing debugging frameworks can pinpoint these failures by identifying key failure attributes, repairing the model effectively remains difficult. Current solutions often rely on manually designed prompts to generate synthetic training images -- an approach prone to distribution shift and semantic errors. To overcome these challenges, we introduce a model repair module that builds on an interpretable failure attribution pipeline. Our approach uses a conditional text-to-image model to generate semantically faithful and targeted images for failure cases. To preserve the quality and relevance of the generated samples, we further employ a large vision-language model (LVLM) to filter the outputs, enforcing alignment with the original data distribution and maintaining semantic consistency. By retraining vision models with this rare-case-augmented synthetic dataset, we significantly reduce errors associated with rare cases. Our experiments demonstrate that this targeted repair strategy improves model robustness without introducing new bugs. Code is available at this https URL 

**Abstract (ZH)**: 基于可解释故障归因管道的模型修复模块：生成语义忠实且针对性的图像以改进视觉模型的鲁棒性 

---
# MMIF-AMIN: Adaptive Loss-Driven Multi-Scale Invertible Dense Network for Multimodal Medical Image Fusion 

**Title (ZH)**: MMIF-AMIN：自适应损失驱动的多尺度可逆密集网络用于多模态医疗图像融合 

**Authors**: Tao Luo, Weihua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08679)  

**Abstract**: Multimodal medical image fusion (MMIF) aims to integrate images from different modalities to produce a comprehensive image that enhances medical diagnosis by accurately depicting organ structures, tissue textures, and metabolic information. Capturing both the unique and complementary information across multiple modalities simultaneously is a key research challenge in MMIF. To address this challenge, this paper proposes a novel image fusion method, MMIF-AMIN, which features a new architecture that can effectively extract these unique and complementary features. Specifically, an Invertible Dense Network (IDN) is employed for lossless feature extraction from individual modalities. To extract complementary information between modalities, a Multi-scale Complementary Feature Extraction Module (MCFEM) is designed, which incorporates a hybrid attention mechanism, convolutional layers of varying sizes, and Transformers. An adaptive loss function is introduced to guide model learning, addressing the limitations of traditional manually-designed loss functions and enhancing the depth of data mining. Extensive experiments demonstrate that MMIF-AMIN outperforms nine state-of-the-art MMIF methods, delivering superior results in both quantitative and qualitative analyses. Ablation experiments confirm the effectiveness of each component of the proposed method. Additionally, extending MMIF-AMIN to other image fusion tasks also achieves promising performance. 

**Abstract (ZH)**: 多模态医疗图像融合（MMIF）旨在通过综合不同模态的图像，生成能够准确展现器官结构、组织纹理和代谢信息的全面图像，从而提升医学诊断。同时捕捉多个模态的独特和互补信息是MMIF的关键研究挑战。为应对这一挑战，本文提出了一种新型图像融合方法MMIF-AMIN，该方法采用了一种新的架构，能够有效提取这些独特和互补的特征。具体而言，本文采用不可逆密集网络（IDN）从单个模态中进行无损特征提取。为了提取模态间互补信息，设计了一种多尺度互补特征提取模块（MCFEM），该模块结合了混合注意力机制、不同大小的卷积层和Transformer。引入了一种自适应损失函数来引导模型学习，克服了传统手动设计损失函数的局限，增强了数据挖掘的深度。大量实验表明，MMIF-AMIN在定量和定性分析中均优于九种最先进的MMIF方法。消融实验验证了所提方法每个组件的有效性。此外，将MMIF-AMIN扩展到其他图像融合任务也取得了令人鼓舞的效果。 

---
# Yan: Foundational Interactive Video Generation 

**Title (ZH)**: 颜: 基础交互式视频生成 

**Authors**: Yan Team  

**Link**: [PDF](https://arxiv.org/pdf/2508.08601)  

**Abstract**: We present Yan, a foundational framework for interactive video generation, covering the entire pipeline from simulation and generation to editing. Specifically, Yan comprises three core modules. AAA-level Simulation: We design a highly-compressed, low-latency 3D-VAE coupled with a KV-cache-based shift-window denoising inference process, achieving real-time 1080P/60FPS interactive simulation. Multi-Modal Generation: We introduce a hierarchical autoregressive caption method that injects game-specific knowledge into open-domain multi-modal video diffusion models (VDMs), then transforming the VDM into a frame-wise, action-controllable, real-time infinite interactive video generator. Notably, when the textual and visual prompts are sourced from different domains, the model demonstrates strong generalization, allowing it to blend and compose the style and mechanics across domains flexibly according to user prompts. Multi-Granularity Editing: We propose a hybrid model that explicitly disentangles interactive mechanics simulation from visual rendering, enabling multi-granularity video content editing during interaction through text. Collectively, Yan offers an integration of these modules, pushing interactive video generation beyond isolated capabilities toward a comprehensive AI-driven interactive creation paradigm, paving the way for the next generation of creative tools, media, and entertainment. The project page is: this https URL. 

**Abstract (ZH)**: 颜：一种全面的交互视频生成基础框架 

---
# Spatiotemporally Consistent Indoor Lighting Estimation with Diffusion Priors 

**Title (ZH)**: 时空一致的室内照明估计基于扩散先验 

**Authors**: Mutian Tong, Rundi Wu, Changxi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08384)  

**Abstract**: Indoor lighting estimation from a single image or video remains a challenge due to its highly ill-posed nature, especially when the lighting condition of the scene varies spatially and temporally. We propose a method that estimates from an input video a continuous light field describing the spatiotemporally varying lighting of the scene. We leverage 2D diffusion priors for optimizing such light field represented as a MLP. To enable zero-shot generalization to in-the-wild scenes, we fine-tune a pre-trained image diffusion model to predict lighting at multiple locations by jointly inpainting multiple chrome balls as light probes. We evaluate our method on indoor lighting estimation from a single image or video and show superior performance over compared baselines. Most importantly, we highlight results on spatiotemporally consistent lighting estimation from in-the-wild videos, which is rarely demonstrated in previous works. 

**Abstract (ZH)**: 从单张图像或视频估计室内光照仍然是一个挑战，尤其是在光照条件在空间和时间上变化的情况下。我们提出了一种方法，通过输入视频估计一个连续的光场，描述场景的时空变化光照。我们利用2D扩散先验优化这种以多层感知机表示的光场。为了使模型能够零样本泛化到真实场景，我们通过同时 inpaint 多个 chrome 球作为光照探针来微调预训练的图像扩散模型，以预测多个位置的光照。我们在单张图像或视频的室内光照估计上评估了该方法，并展示了相对于基线的优越性能。更重要的是，我们在真实视频的时空一致光照估计方面展示了结果，这是先前工作中鲜有展示的内容。 

---
# ImageDDI: Image-enhanced Molecular Motif Sequence Representation for Drug-Drug Interaction Prediction 

**Title (ZH)**: ImageDDI: 基于图像增强的分子motif序列表示方法用于药物-药物相互作用预测 

**Authors**: Yuqin He, Tengfei Ma, Chaoyi Li, Pengsen Ma, Hongxin Xiang, Jianmin Wang, Yiping Liu, Bosheng Song, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08338)  

**Abstract**: To mitigate the potential adverse health effects of simultaneous multi-drug use, including unexpected side effects and interactions, accurately identifying and predicting drug-drug interactions (DDIs) is considered a crucial task in the field of deep learning. Although existing methods have demonstrated promising performance, they suffer from the bottleneck of limited functional motif-based representation learning, as DDIs are fundamentally caused by motif interactions rather than the overall drug structures. In this paper, we propose an Image-enhanced molecular motif sequence representation framework for \textbf{DDI} prediction, called ImageDDI, which represents a pair of drugs from both global and local structures. Specifically, ImageDDI tokenizes molecules into functional motifs. To effectively represent a drug pair, their motifs are combined into a single sequence and embedded using a transformer-based encoder, starting from the local structure representation. By leveraging the associations between drug pairs, ImageDDI further enhances the spatial representation of molecules using global molecular image information (e.g. texture, shadow, color, and planar spatial relationships). To integrate molecular visual information into functional motif sequence, ImageDDI employs Adaptive Feature Fusion, enhancing the generalization of ImageDDI by dynamically adapting the fusion process of feature representations. Experimental results on widely used datasets demonstrate that ImageDDI outperforms state-of-the-art methods. Moreover, extensive experiments show that ImageDDI achieved competitive performance in both 2D and 3D image-enhanced scenarios compared to other models. 

**Abstract (ZH)**: 基于图像增强分子motif序列表示的DDI预测框架：ImageDDI 

---
# Evaluation of State-of-the-Art Deep Learning Techniques for Plant Disease and Pest Detection 

**Title (ZH)**: 基于最新深度学习技术的植物病虫害检测评价 

**Authors**: Saptarshi Banerjee, Tausif Mallick, Amlan Chakroborty, Himadri Nath Saha, Nityananda T. Takur  

**Link**: [PDF](https://arxiv.org/pdf/2508.08317)  

**Abstract**: Addressing plant diseases and pests is critical for enhancing crop production and preventing economic losses. Recent advances in artificial intelligence (AI), machine learning (ML), and deep learning (DL) have significantly improved the precision and efficiency of detection methods, surpassing the limitations of manual identification. This study reviews modern computer-based techniques for detecting plant diseases and pests from images, including recent AI developments. The methodologies are organized into five categories: hyperspectral imaging, non-visualization techniques, visualization approaches, modified deep learning architectures, and transformer models. This structured taxonomy provides researchers with detailed, actionable insights for selecting advanced state-of-the-art detection methods. A comprehensive survey of recent work and comparative studies demonstrates the consistent superiority of modern AI-based approaches, which often outperform older image analysis methods in speed and accuracy. In particular, vision transformers such as the Hierarchical Vision Transformer (HvT) have shown accuracy exceeding 99.3% in plant disease detection, outperforming architectures like MobileNetV3. The study concludes by discussing system design challenges, proposing solutions, and outlining promising directions for future research. 

**Abstract (ZH)**: 植物疾病的防治和害虫管理对于提高作物产量和防止经济损失至关重要。近年来，人工智能（AI）、机器学习（ML）和深度学习（DL）的进展显著提高了检测方法的精确性和效率，超越了人工识别的局限性。本文回顾了现代基于计算机的图像检测植物疾病和害虫的方法，包括最新的AI发展。这些方法被分为五类：超光谱成像、非可视化技术、可视化方法、修改的深度学习架构和.transformer模型。这种结构化的分类体系为研究人员提供了详细的、可操作的见解，以选择先进的先进方法。全面的工作综述和比较研究显示，现代基于AI的方法在速度和准确性方面通常优于较早的图像分析方法，特别是层次视觉变压器（HvT）在植物疾病检测中的准确率超过99.3%，优于MobileNetV3等架构。本文最后讨论了系统设计挑战，提出了解决方案，并概述了未来研究的有希望的方向。 

---
