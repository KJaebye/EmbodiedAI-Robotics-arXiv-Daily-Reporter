# Efficient Virtuoso: A Latent Diffusion Transformer Model for Goal-Conditioned Trajectory Planning 

**Title (ZH)**: 高效的维鲁佐：一种用于目标条件轨迹规划的潜扩散变换器模型 

**Authors**: Antonio Guillen-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03658)  

**Abstract**: The ability to generate a diverse and plausible distribution of future trajectories is a critical capability for autonomous vehicle planning systems. While recent generative models have shown promise, achieving high fidelity, computational efficiency, and precise control remains a significant challenge. In this paper, we present the \textbf{Efficient Virtuoso}, a conditional latent diffusion model for goal-conditioned trajectory planning. Our approach introduces a novel two-stage normalization pipeline that first scales trajectories to preserve their geometric aspect ratio and then normalizes the resulting PCA latent space to ensure a stable training target. The denoising process is performed efficiently in this low-dimensional latent space by a simple MLP denoiser, which is conditioned on a rich scene context fused by a powerful Transformer-based StateEncoder. We demonstrate that our method achieves state-of-the-art performance on the Waymo Open Motion Dataset, reaching a \textbf{minADE of 0.25}. Furthermore, through a rigorous ablation study on goal representation, we provide a key insight: while a single endpoint goal can resolve strategic ambiguity, a richer, multi-step sparse route is essential for enabling the precise, high-fidelity tactical execution that mirrors nuanced human driving behavior. 

**Abstract (ZH)**: 高效虚拟大师：一种条件潜扩散模型在目标条件轨迹规划中的应用 

---
# YOLO Ensemble for UAV-based Multispectral Defect Detection in Wind Turbine Components 

**Title (ZH)**: 基于UAV的风力发电机组件多光谱缺陷检测YOLO集成方法 

**Authors**: Serhii Svystun, Pavlo Radiuk, Oleksandr Melnychenko, Oleg Savenko, Anatoliy Sachenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.04156)  

**Abstract**: Unmanned aerial vehicles (UAVs) equipped with advanced sensors have opened up new opportunities for monitoring wind power plants, including blades, towers, and other critical components. However, reliable defect detection requires high-resolution data and efficient methods to process multispectral imagery. In this research, we aim to enhance defect detection accuracy through the development of an ensemble of YOLO-based deep learning models that integrate both visible and thermal channels. We propose an ensemble approach that integrates a general-purpose YOLOv8 model with a specialized thermal model, using a sophisticated bounding box fusion algorithm to combine their predictions. Our experiments show this approach achieves a mean Average Precision (mAP@.5) of 0.93 and an F1-score of 0.90, outperforming a standalone YOLOv8 model, which scored an mAP@.5 of 0.91. These findings demonstrate that combining multiple YOLO architectures with fused multispectral data provides a more reliable solution, improving the detection of both visual and thermal defects. 

**Abstract (ZH)**: 装备有高级传感器的无人机为监测风力发电厂，包括叶片、塔和其他关键组件，开辟了新机会。然而，可靠的缺陷检测需要高分辨率数据和高效的多光谱图像处理方法。在本研究中，我们通过开发结合可见光和热红外通道的YOLO基集成深度学习模型，旨在提升缺陷检测的准确性。我们提出了一种集成方法，结合了一种通用的YOLOv8模型和一种专门的热红外模型，使用复杂的边界框融合算法来结合它们的预测。实验结果显示，该方法在mAP@.5上的平均精度为0.93，F1分数为0.90，优于单独使用的YOLOv8模型，后者在mAP@.5上的得分为0.91。这些发现表明，结合多个YOLO架构和融合的多光谱数据，提供了更可靠的方法，提高了对视觉和热缺陷的检测能力。 

---
# Expedition & Expansion: Leveraging Semantic Representations for Goal-Directed Exploration in Continuous Cellular Automata 

**Title (ZH)**: 探索与扩展：利用语义表示进行连续细胞自动机中的目标导向探索 

**Authors**: Sina Khajehabdollahi, Gautier Hamon, Marko Cvjetko, Pierre-Yves Oudeyer, Clément Moulin-Frier, Cédric Colas  

**Link**: [PDF](https://arxiv.org/pdf/2509.03863)  

**Abstract**: Discovering diverse visual patterns in continuous cellular automata (CA) is challenging due to the vastness and redundancy of high-dimensional behavioral spaces. Traditional exploration methods like Novelty Search (NS) expand locally by mutating known novel solutions but often plateau when local novelty is exhausted, failing to reach distant, unexplored regions. We introduce Expedition and Expansion (E&E), a hybrid strategy where exploration alternates between local novelty-driven expansions and goal-directed expeditions. During expeditions, E&E leverages a Vision-Language Model (VLM) to generate linguistic goals--descriptions of interesting but hypothetical patterns that drive exploration toward uncharted regions. By operating in semantic spaces that align with human perception, E&E both evaluates novelty and generates goals in conceptually meaningful ways, enhancing the interpretability and relevance of discovered behaviors. Tested on Flow Lenia, a continuous CA known for its rich, emergent behaviors, E&E consistently uncovers more diverse solutions than existing exploration methods. A genealogical analysis further reveals that solutions originating from expeditions disproportionately influence long-term exploration, unlocking new behavioral niches that serve as stepping stones for subsequent search. These findings highlight E&E's capacity to break through local novelty boundaries and explore behavioral landscapes in human-aligned, interpretable ways, offering a promising template for open-ended exploration in artificial life and beyond. 

**Abstract (ZH)**: 探索连续细胞自动机中多样视觉模式的挑战在于高维行为空间的浩瀚和冗余性。传统的探索方法如新颖性搜索（NS）通过突变已知的新颖解决方案进行局部扩展，但在局部新颖性耗尽时往往会停滞，无法到达遥远的未探索区域。我们提出了探险与扩展（E&E）的混合策略，其中探索在局部新颖性驱动的扩展和目标导向的探险之间交替进行。在探险期间，E&E 利用视觉语言模型（VLM）生成语义目标——描述有趣但假设中的模式的描述，从而驱动探索向未开发区域前行。通过在与人类感知相一致的语义空间中操作，E&E 既能评估新颖性，又能以概念上有意义的方式生成目标，从而增强发现行为的可解释性和相关性。在 Flow Lenia 上测试，这是一种以其丰富的涌现行为而闻名的连续 CA，E&E 一致地发现了比现有探索方法更多样化的解决方案。谱系分析进一步表明，起源于探险的解决方案在长期探索中占主导地位，解锁了作为后续搜索阶梯的新行为生态位。这些发现突显了 E&E 有能力突破局部新颖性边界，并以与人类感知一致、可解释的方式探索行为景观，为人工生命及其相关的开放探索提供了有前景的模板。 

---
# SSGaussian: Semantic-Aware and Structure-Preserving 3D Style Transfer 

**Title (ZH)**: SSGaussian: 具有语义意识和结构保真的3D风格迁移 

**Authors**: Jimin Xu, Bosheng Qin, Tao Jin, Zhou Zhao, Zhenhui Ye, Jun Yu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04379)  

**Abstract**: Recent advancements in neural representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have increased interest in applying style transfer to 3D scenes. While existing methods can transfer style patterns onto 3D-consistent neural representations, they struggle to effectively extract and transfer high-level style semantics from the reference style image. Additionally, the stylized results often lack structural clarity and separation, making it difficult to distinguish between different instances or objects within the 3D scene. To address these limitations, we propose a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models. Our pipeline consists of two key stages: First, we leverage diffusion priors to generate stylized renderings of key viewpoints. Then, we transfer the stylized key views onto the 3D representation. This process incorporates two innovative designs. The first is cross-view style alignment, which inserts cross-view attention into the last upsampling block of the UNet, allowing feature interactions across multiple key views. This ensures that the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency. The second is instance-level style transfer, which effectively leverages instance-level consistency across stylized key views and transfers it onto the 3D representation. This results in a more structured, visually coherent, and artistically enriched stylization. Extensive qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods across a wide range of scenes, from forward-facing to challenging 360-degree environments. Visit our project page this https URL for immersive visualization. 

**Abstract (ZH)**: 近年来，神经表示领域的最新进展，如神经光度场和3D高斯斑点化，增加了将风格转移应用于三维场景的兴趣。尽管现有方法可以在3D一致的神经表示上转移风格模式，但在有效提取和转移参考风格图像中的高层次风格语义方面存在局限性。此外，风格化结果往往缺乏结构性清晰度和分离度，使得在三维场景中区分不同的实例或对象变得困难。为了解决这些局限性，我们提出了一种新颖的三维风格转移流水线，有效地整合了预训练二维扩散模型的先验知识。该流水线包含两个关键阶段：首先，我们利用扩散先验生成关键视角的风格化渲染图。然后，将风格化的关键视图转移至三维表示。这一过程包含两个创新设计。第一个是跨视图风格对齐，该设计将跨视图注意力机制插入UNet的最后一个上采样块，允许跨多个关键视图的特征交互。这确保了扩散模型生成的风格化关键视图既能保持风格保真度，又能保持实例级一致性。第二个是实例级风格转移，该设计巧妙地利用了风格化关键视图之间的实例级一致性，并将其转移到三维表示上。这导致了结构更加清晰、视觉上更加连贯、艺术性更强的风格化效果。广泛的定性和定量实验表明，我们的三维风格转移流水线在从面向前方到具有挑战性的360度环境的各种场景中，显著优于现有最先进的方法。请访问我们的项目页面 <https://project.com> 以进行沉浸式可视化。 

---
# From Editor to Dense Geometry Estimator 

**Title (ZH)**: 从编辑者到密集几何估计器 

**Authors**: JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04338)  

**Abstract**: Leveraging visual priors from pre-trained text-to-image (T2I) generative models has shown success in dense prediction. However, dense prediction is inherently an image-to-image task, suggesting that image editing models, rather than T2I generative models, may be a more suitable foundation for fine-tuning.
Motivated by this, we conduct a systematic analysis of the fine-tuning behaviors of both editors and generators for dense geometry estimation. Our findings show that editing models possess inherent structural priors, which enable them to converge more stably by ``refining" their innate features, and ultimately achieve higher performance than their generative counterparts.
Based on these findings, we introduce \textbf{FE2E}, a framework that pioneeringly adapts an advanced editing model based on Diffusion Transformer (DiT) architecture for dense geometry prediction. Specifically, to tailor the editor for this deterministic task, we reformulate the editor's original flow matching loss into the ``consistent velocity" training objective. And we use logarithmic quantization to resolve the precision conflict between the editor's native BFloat16 format and the high precision demand of our tasks. Additionally, we leverage the DiT's global attention for a cost-free joint estimation of depth and normals in a single forward pass, enabling their supervisory signals to mutually enhance each other.
Without scaling up the training data, FE2E achieves impressive performance improvements in zero-shot monocular depth and normal estimation across multiple datasets. Notably, it achieves over 35\% performance gains on the ETH3D dataset and outperforms the DepthAnything series, which is trained on 100$\times$ data. The project page can be accessed \href{this https URL}{here}. 

**Abstract (ZH)**: 利用预训练文本到图像生成模型的视觉先验在密集预测任务中展现了成功。然而，密集预测本质上是图像到图像的任务，表明图像编辑模型而非文本到图像生成模型可能是适合作微调的基础。
受此启发，我们对编辑器和生成器在密集几何估计中的微调行为进行了系统分析。我们的研究结果表明，编辑模型具备内在的结构先验，使其能够通过“优化”其固有特征更稳定地收敛，并最终实现比其生成 counterparts 更高的性能。
基于这些发现，我们引入了\textbf{FE2E}框架，该框架率先将基于扩散变换器（DiT）架构的高级编辑模型适应于密集几何预测。具体而言，为了适应这一确定性任务，我们将编辑器原始的流匹配损失重新表述为“一致速度”训练目标，并使用对数量化来解决编辑器原生的BFloat16格式和任务所需高精度之间的精度冲突。此外，我们利用DiT的全局注意力，在单次前向传播中免费联合估计深度和法线，使它们的监督信号能够相互增强。
在不扩大训练数据的情况下，FE2E在多个数据集中的单目深度和法线估计中实现了显著的性能提升。值得注意的是，它在ETH3D数据集上实现了超过35%的性能提升，并优于DepthAnything系列，该系列模型是在100倍数据上进行训练的。项目页面可访问\href{this https URL}{此处}。 

---
# Learning Active Perception via Self-Evolving Preference Optimization for GUI Grounding 

**Title (ZH)**: 基于自我进化偏好优化的学习主动感知方法及其在GUI定位中的应用 

**Authors**: Wanfu Wang, Qipeng Huang, Guangquan Xue, Xiaobo Liang, Juntao Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04243)  

**Abstract**: Vision Language Models (VLMs) have recently achieved significant progress in bridging visual perception and linguistic reasoning. Recently, OpenAI o3 model introduced a zoom-in search strategy that effectively elicits active perception capabilities in VLMs, improving downstream task performance. However, enabling VLMs to reason effectively over appropriate image regions remains a core challenge in GUI grounding, particularly under high-resolution inputs and complex multi-element visual interactions. In this work, we propose LASER, a self-evolving framework that progressively endows VLMs with multi-step perception capabilities, enabling precise coordinate prediction. Specifically, our approach integrate Monte Carlo quality estimation with Intersection-over-Union (IoU)-based region quality evaluation to jointly encourage both accuracy and diversity in constructing high-quality preference data. This combination explicitly guides the model to focus on instruction-relevant key regions while adaptively allocating reasoning steps based on task complexity. Comprehensive experiments on the ScreenSpot Pro and ScreenSpot-v2 benchmarks demonstrate consistent performance gains, validating the effectiveness of our method. Furthermore, when fine-tuned on GTA1-7B, LASER achieves a score of 55.7 on the ScreenSpot-Pro benchmark, establishing a new state-of-the-art (SoTA) among 7B-scale models. 

**Abstract (ZH)**: Vision-Language Models (VLMs)在视觉感知与语言推理融合方面取得了显著进展。最近，OpenAI的o3模型引入了一种缩放搜索策略，有效激发了VLMs的主动感知能力，提高了下游任务性能。然而，在GUI语义理解中，特别是在高分辨率输入和复杂多元素视觉交互下，使VLMs能够有效地在适当图像区域进行推理仍然是一个核心挑战。在本工作中，我们提出了一种自演化框架LASER，该框架逐步赋予VLMs多步感知能力，实现精确坐标预测。具体而言，我们的方法通过将蒙特卡洛质量估计与基于交并比(IoU)的区域质量评估相结合，共同促进构建高质量偏好数据的准确性和多样性。这种结合明确引导模型关注指令相关的关键区域，并根据任务复杂性自适应分配推理步骤。在ScreenSpot Pro和ScreenSpot-v2基准上的综合性实验显示了持续的性能提升，验证了我们方法的有效性。此外，通过在GTA1-7B上微调后，LASER在ScreenSpot-Pro基准上获得了55.7的分数，成为7B规模模型中的新最佳表现（SoTA）。 

---
# VisioFirm: Cross-Platform AI-assisted Annotation Tool for Computer Vision 

**Title (ZH)**: VisioFirm：跨平台的计算机视觉人工智能辅助标注工具 

**Authors**: Safouane El Ghazouali, Umberto Michelucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.04180)  

**Abstract**: AI models rely on annotated data to learn pattern and perform prediction. Annotation is usually a labor-intensive step that require associating labels ranging from a simple classification label to more complex tasks such as object detection, oriented bounding box estimation, and instance segmentation. Traditional tools often require extensive manual input, limiting scalability for large datasets. To address this, we introduce VisioFirm, an open-source web application designed to streamline image labeling through AI-assisted automation. VisioFirm integrates state-of-the-art foundation models into an interface with a filtering pipeline to reduce human-in-the-loop efforts. This hybrid approach employs CLIP combined with pre-trained detectors like Ultralytics models for common classes and zero-shot models such as Grounding DINO for custom labels, generating initial annotations with low-confidence thresholding to maximize recall. Through this framework, when tested on COCO-type of classes, initial prediction have been proven to be mostly correct though the users can refine these via interactive tools supporting bounding boxes, oriented bounding boxes, and polygons. Additionally, VisioFirm has on-the-fly segmentation powered by Segment Anything accelerated through WebGPU for browser-side efficiency. The tool supports multiple export formats (YOLO, COCO, Pascal VOC, CSV) and operates offline after model caching, enhancing accessibility. VisioFirm demonstrates up to 90\% reduction in manual effort through benchmarks on diverse datasets, while maintaining high annotation accuracy via clustering of connected CLIP-based disambiguate components and IoU-graph for redundant detection suppression. VisioFirm can be accessed from \href{this https URL}{this https URL}. 

**Abstract (ZH)**: AI模型依赖标注数据学习模式和进行预测。标注通常是一个劳动密集型步骤，要求关联从简单分类标签到复杂任务（如对象检测、有向边界框估计和实例分割）的标签。传统工具往往需要大量手动输入，限制了大规模数据集的可扩展性。为了解决这一问题，我们介绍了VisioFirm，一个开源网页应用，旨在通过AI辅助自动化来简化图像标注过程。VisioFirm将最先进的基础模型集成到一个包含过滤管道的界面中，以减少人工在环努力。这种混合方法结合了CLIP与预训练检测器（如Ultralytics模型）以及零样本模型（如Grounding DINO），通过低置信度阈值生成初始标注以最大化召回率。通过该框架，在针对COCO类型类别的测试中，初始预测已被证明大多数是正确的，虽然用户可以通过支持边界框、有向边界框和多边形的交互工具进行进一步精修。此外，VisioFirm通过WebGPU加速的Segment Anything实现了即时分割，增强了浏览器端效率。该工具支持多种导出格式（YOLO、COCO、Pascal VOC、CSV），并通过模型缓存后离线运行，增强了可访问性。VisioFirm通过对连接的CLIP基体消歧组件的聚类和IoU图来减少冗余检测，展示了在多样数据集基准测试中高达90%的手动劳动减少，同时保持高标注准确性。VisioFirm可从\href{this https URL}{this https URL}访问。 

---
# MEPG:Multi-Expert Planning and Generation for Compositionally-Rich Image Generation 

**Title (ZH)**: MEPG：多专家规划与生成在组合丰富的图像生成中的应用 

**Authors**: Yuan Zhao, Liu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04126)  

**Abstract**: Text-to-image diffusion models have achieved remarkable image quality, but they still struggle with complex, multiele ment prompts, and limited stylistic diversity. To address these limitations, we propose a Multi-Expert Planning and Gen eration Framework (MEPG) that synergistically integrates position- and style-aware large language models (LLMs) with spatial-semantic expert modules. The framework comprises two core components: (1) a Position-Style-Aware (PSA) module that utilizes a supervised fine-tuned LLM to decom pose input prompts into precise spatial coordinates and style encoded semantic instructions; and (2) a Multi-Expert Dif fusion (MED) module that implements cross-region genera tion through dynamic expert routing across both local regions and global areas. During the generation process for each lo cal region, specialized models (e.g., realism experts, styliza tion specialists) are selectively activated for each spatial par tition via attention-based gating mechanisms. The architec ture supports lightweight integration and replacement of ex pert models, providing strong extensibility. Additionally, an interactive interface enables real-time spatial layout editing and per-region style selection from a portfolio of experts. Ex periments show that MEPG significantly outperforms base line models with the same backbone in both image quality
and style diversity. 

**Abstract (ZH)**: 基于多专家规划与生成框架的文本到图像扩散模型 

---
# EHVC: Efficient Hierarchical Reference and Quality Structure for Neural Video Coding 

**Title (ZH)**: EHVC:有效分层参考和质量结构的神经视频编码 

**Authors**: Junqi Liao, Yaojun Wu, Chaoyi Lin, Zhipin Deng, Li Li, Dong Liu, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04118)  

**Abstract**: Neural video codecs (NVCs), leveraging the power of end-to-end learning, have demonstrated remarkable coding efficiency improvements over traditional video codecs. Recent research has begun to pay attention to the quality structures in NVCs, optimizing them by introducing explicit hierarchical designs. However, less attention has been paid to the reference structure design, which fundamentally should be aligned with the hierarchical quality structure. In addition, there is still significant room for further optimization of the hierarchical quality structure. To address these challenges in NVCs, we propose EHVC, an efficient hierarchical neural video codec featuring three key innovations: (1) a hierarchical multi-reference scheme that draws on traditional video codec design to align reference and quality structures, thereby addressing the reference-quality mismatch; (2) a lookahead strategy to utilize an encoder-side context from future frames to enhance the quality structure; (3) a layer-wise quality scale with random quality training strategy to stabilize quality structures during inference. With these improvements, EHVC achieves significantly superior performance to the state-of-the-art NVCs. Code will be released in: this https URL. 

**Abstract (ZH)**: 基于神经网络的高效分层次视频编码器（EHVC）：一种结合传统视频编码设计的高效分层次多参考方案、前瞻策略及分层质量尺度的神经视频编解码器 

---
# Neural Video Compression with In-Loop Contextual Filtering and Out-of-Loop Reconstruction Enhancement 

**Title (ZH)**: 基于循环内上下文过滤和循环外重建增强的神经视频压缩 

**Authors**: Yaojun Wu, Chaoyi Lin, Yiming Wang, Semih Esenlik, Zhaobin Zhang, Kai Zhang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04051)  

**Abstract**: This paper explores the application of enhancement filtering techniques in neural video compression. Specifically, we categorize these techniques into in-loop contextual filtering and out-of-loop reconstruction enhancement based on whether the enhanced representation affects the subsequent coding loop. In-loop contextual filtering refines the temporal context by mitigating error propagation during frame-by-frame encoding. However, its influence on both the current and subsequent frames poses challenges in adaptively applying filtering throughout the sequence. To address this, we introduce an adaptive coding decision strategy that dynamically determines filtering application during encoding. Additionally, out-of-loop reconstruction enhancement is employed to refine the quality of reconstructed frames, providing a simple yet effective improvement in coding efficiency. To the best of our knowledge, this work presents the first systematic study of enhancement filtering in the context of conditional-based neural video compression. Extensive experiments demonstrate a 7.71% reduction in bit rate compared to state-of-the-art neural video codecs, validating the effectiveness of the proposed approach. 

**Abstract (ZH)**: 本文探索了增强过滤技术在神经视频压缩中的应用。具体而言，我们根据增强表示是否影响后续编码循环，将这些技术分为循环内上下文过滤和循环外重构增强。循环内上下文过滤通过减轻逐帧编码过程中的误差传播来细化时间上下文，但其对当前帧和后续帧的影响为在整个序列中适应性应用过滤带来了挑战。为此，我们提出了一种自适应编码决策策略，以动态确定编码过程中过滤的应用。此外，循环外重构增强用于细化重建帧的质量，提供了一种简单而有效的编码效率改进方法。据我们所知，本工作是首次对基于条件的神经视频压缩中增强过滤的系统研究。详尽的实验结果表明，相比于最先进的神经视频编解码器，比特率降低了7.71%，验证了所提方法的有效性。 

---
# Detecting Regional Spurious Correlations in Vision Transformers via Token Discarding 

**Title (ZH)**: 基于tokens弃用检测视觉变换器中的区域虚假相关性 

**Authors**: Solha Kang, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem, Joris Vankerschaver, Francois Rameau, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04009)  

**Abstract**: Due to their powerful feature association capabilities, neural network-based computer vision models have the ability to detect and exploit unintended patterns within the data, potentially leading to correct predictions based on incorrect or unintended but statistically relevant signals. These clues may vary from simple color aberrations to small texts within the image. In situations where these unintended signals align with the predictive task, models can mistakenly link these features with the task and rely on them for making predictions. This phenomenon is referred to as spurious correlations, where patterns appear to be associated with the task but are actually coincidental. As a result, detection and mitigation of spurious correlations have become crucial tasks for building trustworthy, reliable, and generalizable machine learning models. In this work, we present a novel method to detect spurious correlations in vision transformers, a type of neural network architecture that gained significant popularity in recent years. Using both supervised and self-supervised trained models, we present large-scale experiments on the ImageNet dataset demonstrating the ability of the proposed method to identify spurious correlations. We also find that, even if the same architecture is used, the training methodology has a significant impact on the model's reliance on spurious correlations. Furthermore, we show that certain classes in the ImageNet dataset contain spurious signals that are easily detected by the models and discuss the underlying reasons for those spurious signals. In light of our findings, we provide an exhaustive list of the aforementioned images and call for caution in their use in future research efforts. Lastly, we present a case study investigating spurious signals in invasive breast mass classification, grounding our work in real-world scenarios. 

**Abstract (ZH)**: 基于神经网络的计算机视觉模型由于其强大的特征关联能力，能够检测和利用数据中的未预期模式， potentially 基于错误或未预期但统计上相关的信号进行正确预测。这些线索可以从简单的颜色偏差到图像内的小型文本不等。当这些未预期的信号与预测任务相一致时，模型可能会错误地将这些特征与任务关联起来并依赖于它们进行预测。这一现象被称为伪相关，即模式看似与任务相关但实际上只是巧合。因此，检测和缓解伪相关已成为构建可信赖、可靠且普适的机器学习模型的关键任务。在本工作中，我们提出了一种新颖的方法来检测视觉变换器中的伪相关，这是一种近年来广受青睐的神经网络架构。我们使用监督训练和自监督训练模型，在ImageNet数据集上进行了大规模实验，展示了所提出方法识别伪相关的能 力。我们还发现，即使使用相同的架构，训练方法对模型依赖伪相关的影响也十分显著。此外，我们展示了ImageNet数据集中的某些类别包含易于被模型检测到的伪信号，并讨论了这些伪信号背后的原因。基于我们的发现，我们列出了上述图像的详尽列表，并呼吁在未来的研究中对此保持警惕。最后，我们通过一个案例研究探讨了侵入性乳腺肿块分类中的伪信号，将我们的工作与实际场景相结合。 

---
# Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer Learning in a U-Net Architecture 

**Title (ZH)**: 使用EfficientNet-B4迁移学习在U-Net架构中的胸片气胸分割 

**Authors**: Alvaro Aranibar Roque, Helga Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2509.03950)  

**Abstract**: Pneumothorax, the abnormal accumulation of air in the pleural space, can be life-threatening if undetected. Chest X-rays are the first-line diagnostic tool, but small cases may be subtle. We propose an automated deep-learning pipeline using a U-Net with an EfficientNet-B4 encoder to segment pneumothorax regions. Trained on the SIIM-ACR dataset with data augmentation and a combined binary cross-entropy plus Dice loss, the model achieved an IoU of 0.7008 and Dice score of 0.8241 on the independent PTX-498 dataset. These results demonstrate that the model can accurately localize pneumothoraces and support radiologists. 

**Abstract (ZH)**: 胸腔积气的自动化深度学习分割pipeline：基于EfficientNet-B4编码器的U-Net模型在SIIM-ACR数据集上的训练及在PTX-498数据集上的独立验证 

---
# SalientFusion: Context-Aware Compositional Zero-Shot Food Recognition 

**Title (ZH)**: SalientFusion: 具有上下文意识的组分零样本食物识别 

**Authors**: Jiajun Song, Xiaoou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03873)  

**Abstract**: Food recognition has gained significant attention, but the rapid emergence of new dishes requires methods for recognizing unseen food categories, motivating Zero-Shot Food Learning (ZSFL). We propose the task of Compositional Zero-Shot Food Recognition (CZSFR), where cuisines and ingredients naturally align with attributes and objects in Compositional Zero-Shot learning (CZSL). However, CZSFR faces three challenges: (1) Redundant background information distracts models from learning meaningful food features, (2) Role confusion between staple and side dishes leads to misclassification, and (3) Semantic bias in a single attribute can lead to confusion of understanding. Therefore, we propose SalientFusion, a context-aware CZSFR method with two components: SalientFormer, which removes background redundancy and uses depth features to resolve role confusion; DebiasAT, which reduces the semantic bias by aligning prompts with visual features. Using our proposed benchmarks, CZSFood-90 and CZSFood-164, we show that SalientFusion achieves state-of-the-art results on these benchmarks and the most popular general datasets for the general CZSL. The code is avaliable at this https URL. 

**Abstract (ZH)**: 食品识别引起了广泛关注，但新菜品的迅速涌现需要能够识别未见过的食品类别的方法，这促进了零样本食品学习（ZSFL）的发展。我们提出了组合零样本食品识别（CZSFR）任务，在组合零样本学习（CZSL）中，烹饪方式和食材自然地与属性和对象相匹配。然而，CZSFR 面临三大挑战：（1）冗余背景信息会干扰模型学习有意义的食品特征，（2）主食与配菜的角色混淆导致分类错误，（3）单个属性中的语义偏见可能导致理解混淆。因此，我们提出了一个基于上下文的 CZSFR 方法——SalientFusion，该方法由两个组件组成：SalientFormer，它去除背景冗余并使用深度特征解决角色混淆；DebiasAT，它通过将提示与视觉特征对齐来减少语义偏见。使用我们提出的基准数据集 CZSFood-90 和 CZSFood-164，我们证明了 SalientFusion 在这些基准及最流行的通用 CZSL 数据集上达到了最先进的性能。代码可在此处访问：https://xxxxxx。 

---
# LuxDiT: Lighting Estimation with Video Diffusion Transformer 

**Title (ZH)**: LuxDiT: 视频扩散变换器的照明 estimation 

**Authors**: Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, Zian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03680)  

**Abstract**: Estimating scene lighting from a single image or video remains a longstanding challenge in computer vision and graphics. Learning-based approaches are constrained by the scarcity of ground-truth HDR environment maps, which are expensive to capture and limited in diversity. While recent generative models offer strong priors for image synthesis, lighting estimation remains difficult due to its reliance on indirect visual cues, the need to infer global (non-local) context, and the recovery of high-dynamic-range outputs. We propose LuxDiT, a novel data-driven approach that fine-tunes a video diffusion transformer to generate HDR environment maps conditioned on visual input. Trained on a large synthetic dataset with diverse lighting conditions, our model learns to infer illumination from indirect visual cues and generalizes effectively to real-world scenes. To improve semantic alignment between the input and the predicted environment map, we introduce a low-rank adaptation finetuning strategy using a collected dataset of HDR panoramas. Our method produces accurate lighting predictions with realistic angular high-frequency details, outperforming existing state-of-the-art techniques in both quantitative and qualitative evaluations. 

**Abstract (ZH)**: 从单张图像或视频估计场景光照仍然是计算机视觉和图形学中的一个长期挑战。基于学习的方法受限于高动态范围环境图的地真数据稀缺，这些数据昂贵且多样性有限。尽管最近的生成模型提供了强大的先验知识用于图像合成，但由于依赖于间接视觉线索、需要推断全局（非局部）上下文以及恢复高动态范围输出，光照估计仍然困难。我们提出LuxDiT，这是一种新颖的数据驱动方法，通过微调视频扩散变换器来生成条件于视觉输入的高动态范围环境图。在包含各种光照条件的大型合成数据集上训练，我们的模型学会了从间接视觉线索中推断照明，并能够有效地泛化到真实场景。为了改进输入和预测环境图之间的语义对齐，我们引入了一种基于收集的高动态范围全景图数据集的低秩适应微调策略。我们的方法在定量和定性评估中均产生准确的光照预测，并具有现实的角方向高频细节，优于现有最先进的技术。 

---
# treeX: Unsupervised Tree Instance Segmentation in Dense Forest Point Clouds 

**Title (ZH)**: treeX：密集森林点云中的无监督树实例分割 

**Authors**: Josafat-Mattias Burmeister, Andreas Tockner, Stefan Reder, Markus Engel, Rico Richter, Jan-Peter Mund, Jürgen Döllner  

**Link**: [PDF](https://arxiv.org/pdf/2509.03633)  

**Abstract**: Close-range laser scanning provides detailed 3D captures of forest stands but requires efficient software for processing 3D point cloud data and extracting individual trees. Although recent studies have introduced deep learning methods for tree instance segmentation, these approaches require large annotated datasets and substantial computational resources. As a resource-efficient alternative, we present a revised version of the treeX algorithm, an unsupervised method that combines clustering-based stem detection with region growing for crown delineation. While the original treeX algorithm was developed for personal laser scanning (PLS) data, we provide two parameter presets, one for ground-based laser scanning (stationary terrestrial - TLS and PLS), and one for UAV-borne laser scanning (ULS). We evaluated the method on six public datasets (FOR-instance, ForestSemantic, LAUTx, NIBIO MLS, TreeLearn, Wytham Woods) and compared it to six open-source methods (original treeX, treeiso, RayCloudTools, ForAINet, SegmentAnyTree, TreeLearn). Compared to the original treeX algorithm, our revision reduces runtime and improves accuracy, with instance detection F$_1$-score gains of +0.11 to +0.49 for ground-based data. For ULS data, our preset achieves an F$_1$-score of 0.58, whereas the original algorithm fails to segment any correct instances. For TLS and PLS data, our algorithm achieves accuracy similar to recent open-source methods, including deep learning. Given its algorithmic design, we see two main applications for our method: (1) as a resource-efficient alternative to deep learning approaches in scenarios where the data characteristics align with the method design (sufficient stem visibility and point density), and (2) for the semi-automatic generation of labels for deep learning models. To enable broader adoption, we provide an open-source Python implementation in the pointtree package. 

**Abstract (ZH)**: 近距离激光扫描提供了森林立地的详细3D捕获，但需要高效的软件来处理3D点云数据并提取单株树。尽管近期的研究引入了深度学习方法进行树实例分割，这些方法需要大型标注数据集和大量的计算资源。作为一种资源高效的替代方案，我们提出了树X算法的一种修订版本，该算法结合基于聚类的主干检测和区域生长的树冠界定，是一种无监督的方法。原树X算法是为个人激光扫描(PLS)数据开发的，我们提供了两个参数预设，一个适用于基于地面激光扫描（固定地面激光扫描-TLS和PLS），另一个适用于无人机搭载激光扫描（ULS）。我们在六个公开数据集（FOR-instance、ForestSemantic、LAUTx、NIBIO MLS、TreeLearn、Wytham Woods）上评估了该方法，并将其与六种开源方法（原始树X、treeiso、RayCloudTools、ForAINet、SegmentAnyTree、TreeLearn）进行了比较。与原始树X算法相比，我们的修订版本减少了运行时间并提高了准确性，对于基于地面的数据，实例检测F₁-分数提高了0.11到0.49。对于ULS数据，我们的预设达到了0.58的F₁-分数，而原始算法无法分割任何正确实例。对于TLS和PLS数据，我们的算法在准确性上类似于最近的开源方法，包括深度学习方法。鉴于其算法设计，我们看到该方法的两个主要应用领域：（1）在数据特征与方法设计相匹配的情况下（充分的主干可见性和点密度），作为深度学习方法的一种资源高效替代方案；（2）用于深度学习模型的半自动标签生成。为了促进更广泛的应用，我们在pointtree包中提供了该方法的开源Python实现。 

---
